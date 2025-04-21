package com.gmibank.stepdefinitions;

import com.gmibank.pojos.Customer;
import com.gmibank.utilities.ConfigurationReader;
import io.cucumber.java.en.*;
import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;
import io.restassured.http.ContentType;
import io.restassured.response.Response;
import org.checkerframework.checker.units.qual.C;
import org.codehaus.jackson.map.ObjectMapper;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;

import static io.restassured.RestAssured.*;

public class US_21_CountryStepDef {

    Response response;
    @Given("manipulate all customers' data {string}")
    public void manipulateAllCustomersData(String CustomerApiUrl) {
             response = given()
                                    .accept(ContentType.JSON)
                                        .auth().oauth2(ConfigurationReader.getProperty("api_bearer_token"))

                                .when()
                                    .get(CustomerApiUrl);
                response.prettyPrint();
    }

    @And("user sets the data in correspondent files")
    public void userSetsTheDataInCorrespondentFiles() throws IOException {
        // javadan jsona deserialize  ObjectMaper  key valueye uygun

        ObjectMapper objectMapper = new ObjectMapper();
        Customer[] allCustomer =  objectMapper.readValue(response.asString(), Customer[].class);
        List<Customer> allCustumerList = Arrays.asList(allCustomer);
        allCustumerList.get(2).getFirstName();
    }

    @And("user read all customers you created and validate them {int} by {int}")
    public void userReadAllCustomersYouCreatedAndValidateThemBy(int arg0, int arg1) {
    }

    @Then("user validates {string} data")
    public void userValidatesData(String arg0) {
    }
}
